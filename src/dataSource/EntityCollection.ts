import { Entity, EntityOptions } from './Entity';
import { Event } from './Event';

/**
 * EntityCollection - an ordered collection of {@link Entity} instances,
 * mirroring Cesium's `EntityCollection` (subset).
 *
 * `collectionChanged` is raised with three array arguments
 * `(added, removed, moved)` whenever entities are added or removed.
 */
export class EntityCollection {
  /**
   * Raised on `add` / `remove` / `removeAll`.
   * Listeners receive `(added: Entity[], removed: Entity[], moved: Entity[])`.
   * `moved` is always empty in this subset (no reordering API).
   */
  readonly collectionChanged: Event<[Entity[], Entity[], Entity[]]> = new Event();

  private _entities: Entity[] = [];
  private _byId: Map<string, Entity> = new Map();

  /** Number of entities in the collection. */
  get length(): number {
    return this._entities.length;
  }

  /**
   * Snapshot of the entities in insertion order.
   * Also iterable directly: `for (const entity of collection) { ... }`.
   */
  get values(): readonly Entity[] {
    return this._entities;
  }

  /**
   * Add an existing entity, or create one from options.
   * @returns the added entity.
   * @throws if an entity with the same id is already in the collection.
   */
  add(entityOrOptions: Entity | EntityOptions): Entity {
    const entity =
      entityOrOptions instanceof Entity ? entityOrOptions : new Entity(entityOrOptions);
    if (this._byId.has(entity.id)) {
      throw new Error(`EntityCollection.add: an entity with id '${entity.id}' already exists`);
    }
    this._entities.push(entity);
    this._byId.set(entity.id, entity);
    this.collectionChanged.raiseEvent([entity], [], []);
    return entity;
  }

  /**
   * Remove an entity from the collection.
   * @returns true if the entity was found and removed.
   */
  remove(entity: Entity): boolean {
    const index = this._entities.indexOf(entity);
    if (index === -1) return false;
    this._entities.splice(index, 1);
    this._byId.delete(entity.id);
    this.collectionChanged.raiseEvent([], [entity], []);
    return true;
  }

  /** Remove all entities. Raises `collectionChanged` once with all removed entities. */
  removeAll(): void {
    if (this._entities.length === 0) return;
    const removed = this._entities.slice();
    this._entities.length = 0;
    this._byId.clear();
    this.collectionChanged.raiseEvent([], removed, []);
  }

  /** Get the entity with the given id, or undefined. */
  getById(id: string): Entity | undefined {
    return this._byId.get(id);
  }

  /** Iterate entities in insertion order. */
  *[Symbol.iterator](): IterableIterator<Entity> {
    yield* this._entities;
  }
}

export default EntityCollection;
